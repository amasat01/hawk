#!/bin/sh
# The one owner of the version of every wheel and every manifest: 0.8.0.dev<N>.
# Usage: sh ci/version.sh          print the version
#        sh ci/version.sh stamp    also write it to ci/VERSION, which a source tarball carries
# N is the number of commits that touch this folder, asked of git where the folder is built; a push of the
# folder to another repository adds one commit there, so N grows with every push that changes it. Without a usable
# git history (a tarball, a shallow clone) the stamped ci/VERSION is read; with neither this stops, by name.
# OSPREY_TEST_DEV_NUMBER replaces N and exists only for the clean-install test, which needs a newer build.
cd "$(dirname "$0")/.." || exit 2
base=0.8.0
if [ -n "$OSPREY_TEST_DEV_NUMBER" ]; then
  n=$OSPREY_TEST_DEV_NUMBER
elif git ls-files --error-unmatch ci/version.sh >/dev/null 2>&1 \
  && [ "$(git rev-parse --is-shallow-repository)" = false ]; then
  n=$(git rev-list --count HEAD -- .)
elif [ -f ci/VERSION ]; then
  cat ci/VERSION
  exit 0
else
  echo "ci/version.sh: no complete git history of this folder and no ci/VERSION; run 'sh ci/version.sh stamp' where the history exists" >&2
  exit 1
fi
version=$base.dev$n
[ "$1" != stamp ] || echo "$version" >ci/VERSION
echo "$version"
