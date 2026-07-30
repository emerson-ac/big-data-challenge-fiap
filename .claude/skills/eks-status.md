---
name: eks-status
description: Verifica o estado completo do ambiente de produção do recsys-api no EKS. Mostra pods, deployment, ingress, URL pública e confirma que a API está respondendo.
---

Execute os comandos abaixo e apresente um resumo claro do estado do ambiente:

## 1. Pods e Deployment

```bash
kubectl get pods -n models -o wide
kubectl get deployment recsys-api -n models
```

## 2. Ingress e URL pública

```bash
kubectl get ingress recsys-api -n models
```

## 3. HPA (auto-scaling)

```bash
kubectl describe hpa recsys-api -n models 2>/dev/null || echo "HPA não encontrado"
```

## 4. Health check da API

```bash
curl -s --max-time 10 https://techchallenger2.pocsarcotech.com/health/status
```

## 5. Modelo em produção no MLflow Registry

```bash
curl -s https://mlflow.pocsarcotech.com/api/2.0/mlflow/registered-models/get?name=recsys_recommender \
  | python3 -c "import json,sys; m=json.load(sys.stdin)['registered_model']; v=m['latest_versions'][0]; print(f'Modelo: {m[\"name\"]}\nVersão: {v[\"version\"]}\nStage: {v[\"current_stage\"]}\nAlias @production: {m[\"aliases\"][0][\"version\"] if m[\"aliases\"] else \"não definido\"}')"
```

## Resumo esperado

Apresente:
- **Pods**: X/X prontos
- **URL**: https://techchallenger2.pocsarcotech.com
- **API**: 🟢 respondendo / 🔴 indisponível
- **Modelo em produção**: nome + versão + stage
- **Eventos recentes** (se houver Warning): `kubectl get events -n models --sort-by='.lastTimestamp' | grep Warning | tail -5`
