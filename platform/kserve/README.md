# KServe (modo RawDeployment)

Instalado via Helm (charts OCI oficiais) pelo `scripts/deploy.sh`:

```
helm upgrade --install kserve-crd oci://ghcr.io/kserve/charts/kserve-crd --version $KSERVE_VERSION
helm upgrade --install kserve     oci://ghcr.io/kserve/charts/kserve     --version $KSERVE_VERSION \
  --set kserve.controller.deploymentMode=RawDeployment
```

## Por que RawDeployment
Evita Knative + Istio. O KServe cria apenas Deployment/Service nativos do
Kubernetes para cada `InferenceService`. Sem scale-to-zero, mas muito menos
peças para operar — ideal para PoC acadêmico.

## Exposição pública
O predictor sobe como um Service `ClusterIP` (`<nome>-predictor`, porta 80).
Publicamos ele com um **Ingress classe `alb`** (ver `models/sklearn-iris/ingress.yaml`),
que o AWS Load Balancer Controller materializa como um **ALB internet-facing**.
A URL pública sai em `kubectl get ingress -n models`.

> Ponto que pode exigir ajuste fino na 1ª subida: dependendo da versão do KServe,
> o controller pode também tentar criar um Ingress próprio. Se aparecer ALB
> duplicado, desabilite a criação automática no ConfigMap `inferenceservice-config`
> (chave `ingress` -> `disableIngressCreation: true`) e mantenha só o Ingress deste repo.
