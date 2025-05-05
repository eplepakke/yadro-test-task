import json
import os
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional
from abc import ABC, abstractmethod


class ClassInfo:
    """Represents a UML class with its attributes."""
    def __init__(self, name: str, is_root: bool, documentation: str, attributes: List[Dict[str, str]]):
        self.name = name
        self.is_root = is_root
        self.documentation = documentation
        self.attributes = attributes


class AggregationInfo:
    """Represents an aggregation relationship between classes."""
    def __init__(self, source: str, target: str, source_multiplicity: str, target_multiplicity: str):
        self.source = source
        self.target = target
        self.source_multiplicity = source_multiplicity
        self.target_multiplicity = target_multiplicity
        self.min: Optional[str] = None
        self.max: Optional[str] = None
        self._parse_multiplicity()

    def _parse_multiplicity(self) -> None:
        """Parse source multiplicity to extract min and max values."""
        if self.source_multiplicity == "1":
            self.min = self.max = "1"
        else:
            min_max = self.source_multiplicity.split("..")
            self.min = min_max[0]
            self.max = min_max[1] if len(min_max) > 1 else min_max[0]


class ModelParser:
    """Parses UML model from XMI file."""
    def __init__(self, xml_file: str):
        self.xml_file = xml_file
        self.classes: Dict[str, ClassInfo] = {}
        self.aggregations: List[AggregationInfo] = []
        self.agg_map: Dict[str, List[AggregationInfo]] = {}

    def parse(self) -> None:
        """Parse XMI file to extract classes and aggregations."""
        try:
            tree = ET.parse(self.xml_file)
            root = tree.getroot()

            # Parse classes
            for class_elem in root.findall(".//Class"):
                class_name = class_elem.get("name")
                self.classes[class_name] = ClassInfo(
                    name=class_name,
                    is_root=class_elem.get("isRoot") == "true",
                    documentation=class_elem.get("documentation", ""),
                    attributes=[
                        {"name": attr.get("name"), "type": attr.get("type")}
                        for attr in class_elem.findall("Attribute")
                    ]
                )

            # Parse and cache aggregations
            for agg_elem in root.findall(".//Aggregation"):
                agg = AggregationInfo(
                    source=agg_elem.get("source"),
                    target=agg_elem.get("target"),
                    source_multiplicity=agg_elem.get("sourceMultiplicity"),
                    target_multiplicity=agg_elem.get("targetMultiplicity")
                )
                self.aggregations.append(agg)
                self.agg_map.setdefault(agg.target, []).append(agg)
        except (ET.ParseError, FileNotFoundError) as e:
            raise ValueError(f"Failed to parse XML file {self.xml_file}: {e}")


class ArtifactGeneratorBase(ABC):
    """Abstract base class for artifact generators."""
    @abstractmethod
    def generate(self) -> None:
        """Generate the artifact."""
        pass


class ConfigXmlGenerator(ArtifactGeneratorBase):
    """Generates config.xml based on UML model."""
    def __init__(self, classes: Dict[str, ClassInfo], agg_map: Dict[str, List[AggregationInfo]], output_file: str):
        self.classes = classes
        self.agg_map = agg_map
        self.output_file = output_file

    def generate(self) -> None:
        """Generate config.xml with hierarchical structure of classes."""
        try:
            root_class = next(c for c, info in self.classes.items() if info.is_root)
            bts = ET.Element(root_class)

            # Add BTS attributes
            for attr in self.classes[root_class].attributes:
                attr_elem = ET.SubElement(bts, attr["name"])
                attr_elem.text = attr["type"]

            # Recursive function to add children
            def add_children(parent_elem: ET.Element, parent_class: str) -> None:
                if parent_class not in self.agg_map:
                    return
                for agg in self.agg_map[parent_class]:
                    child_elem = ET.SubElement(parent_elem, agg.source)
                    for attr in self.classes[agg.source].attributes:
                        attr_elem = ET.SubElement(child_elem, attr["name"])
                        attr_elem.text = attr["type"]
                    add_children(child_elem, agg.source)

            add_children(bts, root_class)

            ET.ElementTree(bts).write(self.output_file, encoding="utf-8", xml_declaration=True)
        except Exception as e:
            raise RuntimeError(f"Failed to generate {self.output_file}: {e}")


class MetaJsonGenerator(ArtifactGeneratorBase):
    """Generates meta.json with metadata about classes."""
    def __init__(self, classes: Dict[str, ClassInfo], aggregations: List[AggregationInfo], agg_map: Dict[str, List[AggregationInfo]], output_file: str):
        self.classes = classes
        self.aggregations = aggregations
        self.agg_map = agg_map
        self.output_file = output_file

    def generate(self) -> None:
        """Generate meta.json with class metadata and relationships."""
        try:
            meta = []
            multiplicity_map = {agg.source: {"min": agg.min, "max": agg.max} for agg in self.aggregations}

            for class_name, info in self.classes.items():
                meta_entry = {
                    "class": class_name,
                    "documentation": info.documentation,
                    "isRoot": info.is_root,
                    "parameters": []
                }
                if class_name in multiplicity_map:
                    meta_entry["min"] = multiplicity_map[class_name]["min"]
                    meta_entry["max"] = multiplicity_map[class_name]["max"]

                # Add attributes
                meta_entry["parameters"].extend(
                    {"name": attr["name"], "type": attr["type"]} for attr in info.attributes
                )

                # Add child classes
                meta_entry["parameters"].extend(
                    {"name": agg.source, "type": "class"}
                    for agg in self.aggregations
                    if agg.target == class_name and agg.source != class_name
                )

                meta.append(meta_entry)

            with open(self.output_file, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=4, ensure_ascii=False)
        except Exception as e:
            raise RuntimeError(f"Failed to generate {self.output_file}: {e}")


class ConfigDeltaGenerator(ArtifactGeneratorBase):
    """Generates delta.json and res_patched_config.json."""
    def __init__(self, config: Dict, patched_config: Dict, delta_output_file: str, res_patched_output_file: str):
        self.config = config
        self.patched_config = patched_config
        self.delta_output_file = delta_output_file
        self.res_patched_output_file = res_patched_output_file

    def generate(self) -> None:
        """Generate delta.json and res_patched_config.json based on config differences."""
        try:
            delta = {"additions": [], "deletions": [], "updates": []}
            res_patched = self.config.copy()

            # Process additions and updates
            for key, value in self.patched_config.items():
                if key not in self.config:
                    delta["additions"].append({"key": key, "value": value})
                    res_patched[key] = value
                elif self.config[key] != value:
                    delta["updates"].append({"key": key, "from": self.config[key], "to": value})
                    res_patched[key] = value

            # Process deletions
            for key in self.config:
                if key not in self.patched_config:
                    delta["deletions"].append(key)
                    res_patched.pop(key, None)

            # Write outputs
            with open(self.delta_output_file, "w", encoding="utf-8") as f:
                json.dump(delta, f, indent=4, ensure_ascii=False)
            with open(self.res_patched_output_file, "w", encoding="utf-8") as f:
                json.dump(res_patched, f, indent=4, ensure_ascii=False)
        except Exception as e:
            raise RuntimeError(f"Failed to generate delta or res_patched_config: {e}")


class ArtifactOrchestrator:
    """Coordinates generation of all artifacts."""
    def __init__(self, xml_file: str, config_file: str, patched_config_file: str, output_dir: str):
        self.xml_file = xml_file
        self.config_file = config_file
        self.patched_config_file = patched_config_file
        self.output_dir = output_dir
        self.generators: List[ArtifactGeneratorBase] = []

    def add_generator(self, generator: ArtifactGeneratorBase) -> None:
        """Add a generator to the pipeline."""
        self.generators.append(generator)

    def run(self) -> None:
        """Run all registered generators."""
        try:
            os.makedirs(self.output_dir, exist_ok=True)

            # Parse model
            parser = ModelParser(self.xml_file)
            parser.parse()

            # Load config files
            with open(self.config_file, "r", encoding="utf-8") as f:
                config = json.load(f)
            with open(self.patched_config_file, "r", encoding="utf-8") as f:
                patched_config = json.load(f)

            # Register generators
            self.add_generator(ConfigXmlGenerator(
                parser.classes, parser.agg_map, f"{self.output_dir}/config.xml"
            ))
            self.add_generator(MetaJsonGenerator(
                parser.classes, parser.aggregations, parser.agg_map, f"{self.output_dir}/meta.json"
            ))
            self.add_generator(ConfigDeltaGenerator(
                config, patched_config, f"{self.output_dir}/delta.json", f"{self.output_dir}/res_patched_config.json"
            ))

            # Run generators
            for generator in self.generators:
                generator.generate()
        except Exception as e:
            raise RuntimeError(f"Artifact generation failed: {e}")


def main() -> None:
    """Main entry point for artifact generation."""
    orchestrator = ArtifactOrchestrator(
        xml_file="input/impulse_test_input.xml",
        config_file="input/config.json",
        patched_config_file="input/patched_config.json",
        output_dir="out"
    )
    orchestrator.run()


if __name__ == "__main__":
    main()