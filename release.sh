#!/bin/bash

set -e
package=$1
version=$2

if [[ -z "$package" || -z "$version" ]]; then
  echo "Usage: release.sh <package> <version>"
  exit 1
fi
if [[ ! -d $package ]]; then
  echo "Package $package does not exist"
  exit 1
fi

confirm="n"
read -r -p "Release $package $version? [y/N] " confirm
if [[ ! $confirm =~ ^[yY]$ ]]; then
  echo "Aborted"
  exit 1
fi

cd $package
uvx --from=toml-cli toml set --toml-path=pyproject.toml project.version $version
git commit -am "Release $package $version"
git tag $package-$version
git push origin main
git push origin $package-$version
echo "Release $package $version complete"



