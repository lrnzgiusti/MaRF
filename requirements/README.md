## Requirements README
1. ngp_requirements
    - used by Dockerfile to prepare base environment for initial Instant NGP installation

2. pip_requirements
    - used by Dockerfile to prepare Instant NGP environment for Mars-Metaverse installation
    - generated with pip freeze on development EC2 instance

3. conda_requirements
    - supplemental requirements for Conda users
    - generated with conda list --export
