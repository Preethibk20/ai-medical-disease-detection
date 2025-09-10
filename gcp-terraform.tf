# Google Cloud Platform deployment configuration for AI Medical Disease Detection System

# Variables
variable "project_id" {
  description = "The GCP project ID"
  type        = string
}

variable "region" {
  description = "The GCP region"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "The GCP zone"
  type        = string
  default     = "us-central1-a"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}

variable "machine_type" {
  description = "Machine type for the instance"
  type        = string
  default     = "e2-medium"
}

variable "disk_size" {
  description = "Boot disk size in GB"
  type        = number
  default     = 50
}

variable "disk_type" {
  description = "Boot disk type"
  type        = string
  default     = "pd-ssd"
}

# Provider configuration
terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 4.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

# VPC Network
resource "google_compute_network" "medical_ai_vpc" {
  name                    = "${var.environment}-medical-ai-vpc"
  auto_create_subnetworks = false
  mtu                     = 1460
}

resource "google_compute_subnetwork" "medical_ai_subnet" {
  name          = "${var.environment}-medical-ai-subnet"
  ip_cidr_range = "10.0.1.0/24"
  region        = var.region
  network       = google_compute_network.medical_ai_vpc.id
}

# Firewall rules
resource "google_compute_firewall" "medical_ai_firewall" {
  name    = "${var.environment}-medical-ai-firewall"
  network = google_compute_network.medical_ai_vpc.name

  allow {
    protocol = "tcp"
    ports     = ["22", "80", "443", "8000", "8501"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["medical-ai-server"]
}

# Compute instance
resource "google_compute_instance" "medical_ai_vm" {
  name         = "${var.environment}-medical-ai-vm"
  machine_type = var.machine_type
  zone         = var.zone

  tags = ["medical-ai-server"]

  boot_disk {
    initialize_params {
      image = "ubuntu-os-cloud/ubuntu-2004-lts"
      size  = var.disk_size
      type  = var.disk_type
    }
  }

  network_interface {
    network    = google_compute_network.medical_ai_vpc.name
    subnetwork = google_compute_subnetwork.medical_ai_subnet.name

    access_config {
      // Ephemeral public IP
    }
  }

  metadata = {
    startup-script = <<-EOF
      #!/bin/bash
      
      # Update system
      apt-get update
      apt-get upgrade -y
      
      # Install Docker
      curl -fsSL https://get.docker.com -o get-docker.sh
      sh get-docker.sh
      usermod -aG docker $USER
      
      # Install Docker Compose
      curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
      chmod +x /usr/local/bin/docker-compose
      
      # Install Python and pip
      apt-get install -y python3 python3-pip git curl
      
      # Install Google Cloud SDK
      curl https://sdk.cloud.google.com | bash
      exec -l $SHELL
      
      # Clone application
      cd /home/ubuntu
      sudo -u ubuntu git clone https://github.com/your-repo/medical-ai-system.git
      cd medical-ai-system
      
      # Set environment
      sudo -u ubuntu echo "ENVIRONMENT=${var.environment}" > .env
      sudo -u ubuntu echo "MODEL_CACHE=true" >> .env
      sudo -u ubuntu echo "SECURITY_LEVEL=high" >> .env
      sudo -u ubuntu echo "GCP_PROJECT_ID=${var.project_id}" >> .env
      
      # Start application
      sudo -u ubuntu docker-compose up -d
      
      # Enable logging
      echo "Medical AI System started successfully" >> /var/log/medical-ai-startup.log
    EOF
  }

  service_account {
    email  = google_service_account.medical_ai_sa.email
    scopes = ["cloud-platform"]
  }
}

# Service Account
resource "google_service_account" "medical_ai_sa" {
  account_id   = "${var.environment}-medical-ai-sa"
  display_name = "Medical AI Service Account"
}

# IAM bindings
resource "google_project_iam_member" "medical_ai_storage_admin" {
  project = var.project_id
  role    = "roles/storage.admin"
  member  = "serviceAccount:${google_service_account.medical_ai_sa.email}"
}

resource "google_project_iam_member" "medical_ai_logging_writer" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.medical_ai_sa.email}"
}

# Cloud Storage bucket for models
resource "google_storage_bucket" "medical_ai_models" {
  name          = "${var.environment}-medical-ai-models-${var.project_id}"
  location      = var.region
  force_destroy = true

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      age = 30
    }
    action {
      type = "Delete"
    }
  }
}

# Cloud SQL instance (optional)
resource "google_sql_database_instance" "medical_ai_db" {
  count            = var.environment == "production" ? 1 : 0
  name             = "${var.environment}-medical-ai-db"
  database_version = "POSTGRES_13"
  region           = var.region

  settings {
    tier = "db-f1-micro"
    
    backup_configuration {
      enabled    = true
      start_time = "03:00"
    }
    
    ip_configuration {
      ipv4_enabled = true
      authorized_networks {
        value = "0.0.0.0/0"
      }
    }
  }
}

resource "google_sql_database" "medical_ai_database" {
  count    = var.environment == "production" ? 1 : 0
  name     = "medical_ai"
  instance = google_sql_database_instance.medical_ai_db[0].name
}

resource "google_sql_user" "medical_ai_user" {
  count    = var.environment == "production" ? 1 : 0
  name     = "medical_ai_user"
  instance = google_sql_database_instance.medical_ai_db[0].name
  password = "change-this-password"
}

# Cloud Load Balancer
resource "google_compute_global_address" "medical_ai_ip" {
  count  = var.environment == "production" ? 1 : 0
  name   = "${var.environment}-medical-ai-ip"
  region = var.region
}

resource "google_compute_backend_service" "medical_ai_backend" {
  count       = var.environment == "production" ? 1 : 0
  name        = "${var.environment}-medical-ai-backend"
  protocol    = "HTTP"
  port_name   = "http"
  timeout_sec = 10

  backend {
    group = google_compute_instance_group.medical_ai_group[0].id
  }

  health_checks = [google_compute_health_check.medical_ai_health_check[0].id]
}

resource "google_compute_instance_group" "medical_ai_group" {
  count = var.environment == "production" ? 1 : 0
  name  = "${var.environment}-medical-ai-group"
  zone  = var.zone

  instances = [google_compute_instance.medical_ai_vm.id]

  named_port {
    name = "http"
    port = 80
  }
}

resource "google_compute_health_check" "medical_ai_health_check" {
  count = var.environment == "production" ? 1 : 0
  name  = "${var.environment}-medical-ai-health-check"

  http_health_check {
    port         = 80
    request_path = "/health"
  }
}

resource "google_compute_url_map" "medical_ai_url_map" {
  count           = var.environment == "production" ? 1 : 0
  name            = "${var.environment}-medical-ai-url-map"
  default_service = google_compute_backend_service.medical_ai_backend[0].id
}

resource "google_compute_target_http_proxy" "medical_ai_proxy" {
  count   = var.environment == "production" ? 1 : 0
  name    = "${var.environment}-medical-ai-proxy"
  url_map = google_compute_url_map.medical_ai_url_map[0].id
}

resource "google_compute_global_forwarding_rule" "medical_ai_forwarding_rule" {
  count      = var.environment == "production" ? 1 : 0
  name       = "${var.environment}-medical-ai-forwarding-rule"
  target     = google_compute_target_http_proxy.medical_ai_proxy[0].id
  port_range = "80"
  ip_address = google_compute_global_address.medical_ai_ip[0].address
}

# Outputs
output "instance_external_ip" {
  description = "External IP address of the instance"
  value       = google_compute_instance.medical_ai_vm.network_interface[0].access_config[0].nat_ip
}

output "load_balancer_ip" {
  description = "IP address of the load balancer"
  value       = var.environment == "production" ? google_compute_global_address.medical_ai_ip[0].address : null
}

output "storage_bucket_name" {
  description = "Name of the storage bucket"
  value       = google_storage_bucket.medical_ai_models.name
}

output "database_connection_name" {
  description = "Connection name of the database"
  value       = var.environment == "production" ? google_sql_database_instance.medical_ai_db[0].connection_name : null
}
