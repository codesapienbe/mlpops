terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# Get Ubuntu AMI
data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"]
  
  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-22.04-amd64-server-*"]
  }
}

# Security Group
resource "aws_security_group" "mlops" {
  name_prefix = "mlops-simple-"
  
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  ingress {
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  ingress {
    from_port   = 8501
    to_port     = 8501
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# EC2 Instance (Free Tier)
resource "aws_instance" "mlops" {
  ami           = data.aws_ami.ubuntu.id
  instance_type = "t2.micro"
  
  vpc_security_group_ids = [aws_security_group.mlops.id]
  
  user_data = base64encode(<<-EOF
    #!/bin/bash
    apt-get update
    apt-get install -y docker.io docker-compose git
    systemctl start docker
    systemctl enable docker
    usermod -aG docker ubuntu
    
    # Clone your repo (replace with your actual repo)
    cd /home/ubuntu
    git clone https://github.com/codesapienbe/mlpops.git
    cd mlops-exam
    chown -R ubuntu:ubuntu /home/ubuntu/mlpops
    
    # Start services
    sudo -u ubuntu docker-compose up -d
  EOF
  )
  
  tags = {
    Name = "mlops-simple"
  }
}

output "instance_ip" {
  value = aws_instance.mlops.public_ip
}

output "api_url" {
  value = "http://${aws_instance.mlops.public_ip}:8000"
}

output "web_url" {
  value = "http://${aws_instance.mlops.public_ip}:8501"
}
