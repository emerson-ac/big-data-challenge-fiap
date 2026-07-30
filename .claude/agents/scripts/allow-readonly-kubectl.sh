#!/bin/bash
# Hook PreToolUse para o agent infra-health.
# Bloqueia comandos destrutivos — permite apenas leitura (kubectl get/describe/logs, aws describe/list, curl).
set -euo pipefail

INPUT=$(cat)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // empty')

if [ -z "$COMMAND" ]; then
  exit 0
fi

# Bloquear operações destrutivas no kubectl
if echo "$COMMAND" | grep -E '\bkubectl\b' > /dev/null; then
  if echo "$COMMAND" | grep -iE '\b(delete|apply|patch|replace|scale|rollout restart|rollout undo|edit|create|annotate|label|taint|cordon|drain|uncordon|exec|port-forward|cp)\b' > /dev/null; then
    echo "Bloqueado: o agent infra-health é somente leitura. Operações destrutivas não são permitidas." >&2
    exit 2
  fi
fi

# Bloquear operações destrutivas na AWS
if echo "$COMMAND" | grep -E '\baws\b' > /dev/null; then
  if echo "$COMMAND" | grep -iE '\b(create|delete|update|put|modify|attach|detach|terminate|stop|start|reboot|run|launch|associate|disassociate|replace|authorize|revoke|enable|disable)\b' > /dev/null; then
    echo "Bloqueado: o agent infra-health só pode executar operações de leitura na AWS (describe/list/get)." >&2
    exit 2
  fi
fi

exit 0
