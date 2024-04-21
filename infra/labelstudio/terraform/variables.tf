variable "kube_config" {
  type    = string
  default = "~/.kube/config"
}

variable "namespace" {
  type    = string
  default = "swissimage-vision-labelstudio"
}
