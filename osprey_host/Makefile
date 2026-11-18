# The stateless gate of the OSPREY libraries. `make gate` runs every leg; `make gate-<leg>` runs one
# (fmt clippy test exports binding py39 abi wheels ruff cformat tidy replay). PYTHON is the interpreter the binding is built for;
# CLANG_TIDY names clang-tidy (the tidy leg is red when it is unset). JOBS=2 by default.
JOBS ?= 2
LEGS = fmt clippy test exports binding py39 abi wheels ruff cformat tidy replay

.PHONY: gate $(addprefix gate-,$(LEGS))
gate:
	JOBS=$(JOBS) sh ci/gate.sh

$(addprefix gate-,$(LEGS)): gate-%:
	JOBS=$(JOBS) sh ci/gate.sh $*
