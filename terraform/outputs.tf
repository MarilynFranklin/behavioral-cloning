output "public_ip" {
  value = "${aws_instance.gpu.public_ip}"
}
output "public_dns" {
  value = "${aws_instance.gpu.public_dns}"
}
