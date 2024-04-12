resource "kubernetes_namespace" "labelstudio" {
  metadata {
    name = var.namespace
  }
}


resource "helm_release" "labelstudio" {
  name       = "labelstudio"
  repository = "https://charts.heartex.com/"
  chart      = "label-studio"
  namespace  = var.namespace

  values = ["${file("${path.module}/values.yaml")}"]
}
