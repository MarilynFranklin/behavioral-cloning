data "template_file" "user_data" {
  template = "${file("terraform/user-data.tpl")}"
}

data "aws_ami" "udacity" {
  most_recent = true
  filter {
    name = "name"
    values = ["udacity-carnd"]
  }
}

resource "aws_instance" "gpu" {
  ami = "${data.aws_ami.udacity.id}"
  instance_type = "g2.2xlarge"
  connection {
    type     = "ssh"
    user     = "carnd"
    password = "carnd"
  }
  provisioner "file" {
    source      = "${path.module}/../model.py"
    destination = "/home/carnd/model.py"
  }
  provisioner "file" {
    source      = "${path.module}/../data.zip"
    destination = "/home/carnd/data.zip"
  }
  associate_public_ip_address = true
  vpc_security_group_ids = ["${aws_security_group.allow_traffic.id}"]
  user_data = "${data.template_file.user_data.rendered}"
}
