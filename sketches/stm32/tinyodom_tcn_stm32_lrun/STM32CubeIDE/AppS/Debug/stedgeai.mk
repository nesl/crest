################################################################################
# Repo-maintained ST Edge AI path overrides for CLI builds.
################################################################################

STEDGEAI_CANDIDATES := $(patsubst %/Middlewares/ST/AI/Inc,%,$(sort $(wildcard /opt/ST/STEdgeAI/*/Middlewares/ST/AI/Inc)))
STEDGEAI_ROOT ?= $(lastword $(STEDGEAI_CANDIDATES))
STEDGEAI_INC := $(STEDGEAI_ROOT)/Middlewares/ST/AI/Inc
STEDGEAI_RUNTIME_LIB := $(STEDGEAI_ROOT)/Middlewares/ST/AI/Lib/GCC/ARMCortexM55/NetworkRuntime1200_CM55_GCC.a

ifeq ($(wildcard $(STEDGEAI_INC)),)
$(error ST Edge AI headers not found under "$(STEDGEAI_INC)". Set STEDGEAI_ROOT to your ST Edge AI install root)
endif

ifeq ($(wildcard $(STEDGEAI_RUNTIME_LIB)),)
$(error ST Edge AI runtime library not found at "$(STEDGEAI_RUNTIME_LIB)". Set STEDGEAI_ROOT to your ST Edge AI install root)
endif
