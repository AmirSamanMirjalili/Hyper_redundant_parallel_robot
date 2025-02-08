# Stewart Platform URDF Generator

A Python-based tool for generating URDF (Unified Robot Description Format) files for Stewart Platform robots with multiple stages. This project enables the creation and manipulation of Stewart Platform configurations for simulation in PyBullet and ROS environments.

## Features

- Generate complete URDF descriptions for Stewart Platform robots
- Support for multi-stage configurations
- Automatic generation of:
  - Base and bottom links
  - Cylinder connections
  - Rod and piston assemblies
  - Top platform links
  - Universal joints
- PyBullet-compatible joint configurations
- Proper mesh file handling
- Comprehensive test suite for URDF validation

## Prerequisites

- Python 3.8 or higher
- PyBullet
- XML parsing libraries (xml.etree.ElementTree)
- Python virtual environment (recommended)

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd Stewart-Platform-main
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Generating URDF Files

```python
from generate_urdf import URDFGenerator

# Initialize the generator
generator = URDFGenerator()

# Generate URDF for a specific stage
urdf_content = generator.generate_stage(stage_number=1)

# Save to file
with open('output.urdf', 'w') as f:
    f.write(urdf_content)
```

### Running Tests

```bash
python -m pytest test_urdf.py
```

## Project Structure

- `generate_urdf.py`: Main URDF generation logic
- `extract_urdf_properties.py`: Utilities for extracting properties from existing URDF files
- `test_urdf.py`: Comprehensive test suite
- `Stewart.urdf`: Original reference URDF file
- `Link_graph.txt`: Connection hierarchy documentation
- `meshes/`: Directory containing STL mesh files

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m '[Cursor] Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Original Stewart Platform design contributors
- PyBullet development team
- ROS community for URDF format specifications