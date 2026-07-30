---
name: promote-model
description: Consulta versões disponíveis do recsys_recommender no MLflow Registry externo, exibe métricas e promove uma versão específica para o alias @production.
---

## Passo 1 — Listar versões disponíveis

```bash
curl -s "https://mlflow.pocsarcotech.com/api/2.0/mlflow/model-versions/search?filter=name%3D'recsys_recommender'&max_results=10" \
  | python3 -c "
import json, sys
data = json.load(sys.stdin)
versions = data.get('model_versions', [])
if not versions:
    print('Nenhuma versão encontrada.')
    sys.exit(0)
print(f'{'Versão':<8} {'Stage':<12} {'Run ID':<36} {'Criado em'}')
print('-' * 80)
for v in sorted(versions, key=lambda x: int(x['version'])):
    import datetime
    ts = int(v['creation_timestamp']) // 1000
    dt = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M')
    print(f'{v[\"version\"]:<8} {v[\"current_stage\"]:<12} {v[\"run_id\"]:<36} {dt}')
"
```

## Passo 2 — Buscar métricas de cada versão

Para cada versão listada, busque as métricas do run associado:

```bash
# Substitua <RUN_ID> pelo run_id da versão desejada
curl -s "https://mlflow.pocsarcotech.com/api/2.0/mlflow/runs/get?run_id=<RUN_ID>" \
  | python3 -c "
import json, sys
run = json.load(sys.stdin)['run']
metrics = {m['key']: m['value'] for m in run['data'].get('metrics', [])}
params = {p['key']: p['value'] for p in run['data'].get('params', [])}
tags = {t['key']: t['value'] for t in run['data'].get('tags', [])}
model = tags.get('mlflow.runName', 'desconhecido')
print(f'Modelo: {model}')
for k in ['recall_at_10', 'ndcg_at_10', 'precision_at_10', 'coverage']:
    if k in metrics:
        print(f'  {k}: {metrics[k]:.4f}')
"
```

## Passo 3 — Confirmar e promover

**Pergunte ao usuário** qual versão deseja promover para `@production` antes de executar.

Após confirmação, execute:

```bash
# Substitua <VERSAO> pelo número da versão escolhida
curl -s -X PATCH "https://mlflow.pocsarcotech.com/api/2.0/mlflow/registered-models/alias" \
  -H "Content-Type: application/json" \
  -d '{"name": "recsys_recommender", "alias": "production", "version": "<VERSAO>"}' \
  | python3 -c "import json,sys; r=json.load(sys.stdin); print('✅ Alias @production atualizado para versão', r.get('registered_model_alias', {}).get('version', '?'))"
```

## Passo 4 — Confirmar promoção

```bash
curl -s "https://mlflow.pocsarcotech.com/api/2.0/mlflow/registered-models/alias?name=recsys_recommender&alias=production" \
  | python3 -c "import json,sys; a=json.load(sys.stdin)['registered_model_alias']; print(f'@production → versão {a[\"version\"]}')"
```

## Observação

Após promover, os pods em produção **não recarregam o modelo automaticamente** — o modelo é carregado no startup. Para aplicar imediatamente:

```bash
kubectl rollout restart deployment/recsys-api -n models
kubectl rollout status deployment/recsys-api -n models --timeout=120s
```
