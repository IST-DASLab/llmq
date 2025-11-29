#!/usr/bin/env python3
# Sigh1 pyproject.toml does not provide nice support for dynamic dependencies;
# But we want cuda-12.8 built packages to require 12.8, and 13.0 to require 13.0.
# Seems reasonable, but apparently too much of a niche case for PyPy ?!
# So the solution is to edit the pyproject.toml as part of the build script in the
# workflow :(
import sys
import tomlkit

def add_dependencies(deps_str, cuda_tag=None):
    # Parse dependencies from input
    deps = [dep.strip() for dep in deps_str.strip().split('\n') if dep.strip()]

    with open('pyproject.toml', 'r') as f:
        data = tomlkit.load(f)

    # Add CUDA dependencies
    for dep in deps:
        data['project']['dependencies'].append(dep)

    # Add CUDA tag to version if provided
    if cuda_tag:
        current_version = data['project']['version']
        data['project']['version'] = f"{current_version}+{cuda_tag}"

    with open('pyproject.toml', 'w') as f:
        tomlkit.dump(data, f)

    print(f"Added {len(deps)} CUDA dependencies to pyproject.toml")
    if cuda_tag:
        print(f"Set version to {data['project']['version']}")

if __name__ == '__main__':
    deps = sys.argv[1] if len(sys.argv) > 1 else ''
    cuda_tag = sys.argv[2] if len(sys.argv) > 2 else None
    add_dependencies(deps, cuda_tag)
