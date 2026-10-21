
from pathlib import Path
import xml.etree.ElementTree as ET
from collections import Counter

root_dir = Path("dataset/label")
counts = Counter()

for xml_file in root_dir.rglob("*.xml"):
    root = ET.parse(xml_file).getroot()

    for obj in root.findall("object"):
        name = obj.findtext("name")
        if name:
            counts[name.strip()] += 1

print("Defect annotation counts:\n")

for name, count in sorted(counts.items()):
    print(f"{name:20s} {count:5d}")

print(f"\nTotal annotated objects: {sum(counts.values())}")
