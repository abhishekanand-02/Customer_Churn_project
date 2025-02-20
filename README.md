# Customer_Churn_project - MLOps Pipeline Setup with AWS 🚀

In my current MLOps pipeline, I've set up an automated deployment workflow using AWS services. Let me walk you through the key components and why I chose them:

## 1. GitHub & GitHub Actions 🧑‍💻🤖

### Why GitHub? 
GitHub is our version control system where we store the source code for the machine learning model. It allows us to manage, track, and collaborate on code efficiently. 🔄

### Why GitHub Actions? 
I use GitHub Actions for Continuous Integration and Continuous Deployment (CI/CD). It automates the workflow so that when I push new code to the repository, it triggers actions to build and deploy the model automatically. This removes manual intervention and speeds up the process. ⚙️✨

## 2. Amazon EC2 (Elastic Compute Cloud) 🌥️💻

### Why EC2? 
EC2 is the compute resource where we run our model in the cloud. It's flexible and scalable, so whether we need a small instance for testing or a larger instance for training, we can adjust based on requirements. This helps us avoid investing in expensive on-prem hardware. 💡💵

## 3. Amazon ECR (Elastic Container Registry) 📦🛠️

### Why ECR? 
ECR is where we store the Docker images of our machine learning models. Using Docker ensures that our environment is consistent across development, testing, and production, which is crucial for reproducibility. Once our model is containerized, we push it to ECR, making it easy to deploy on EC2. 🐳✅

## 4. IAM (Identity and Access Management) 🔐

### Why IAM? 
IAM provides secure access control across our AWS resources. By using IAM users and roles, we ensure that only authorized individuals and services can interact with the AWS infrastructure. It’s essential for maintaining security and governance while enabling smooth operations. 🛡️🚪

---

## In Summary 📝:

This setup allows us to automate the deployment of machine learning models in a reproducible, scalable, and secure manner. GitHub Actions helps us with continuous deployment, EC2 gives us the flexibility for compute, ECR ensures consistent deployment via Docker, and IAM secures our cloud environment. This integration streamlines our workflow, reduces human errors, and accelerates our ability to deliver machine learning models into production. ⚡💼

---

## Steps to Setup GitHub Actions, EC2, ECR, and IAM Users ⚙️💻

### 1. **Set Up IAM User 🔑:**

- Go to **IAM Users** section.
- Click on **Add user**.
- Name the user as `mlproj-user`.
- Click **Next**.
- **Attach Policies Directly**:
  - Select `AmazonEC2ContainerRegistryFullAccess`.
  - Select `AmazonEC2FullAccess`.
- Click **Create User**.
- Go to the user details page and click on `mlproj-user`.
- Under **Security Credentials**, click **Create Access Key**.
  - Select **CLI** and click **Next**.
  - Download the `.csv` file containing your access key.

### 2. **Create ECR Repository 📦:**

- In the **ECR** section, click **Create repository**.
- Name the repository as `mlproj`.
- Click **Create repository**.
- Copy the **URI** of the repository.

### 3. **Launch EC2 Instance ☁️💻:**

- Go to **EC2** section and click **Launch Instance**.
- Select the instance name as `mlproj-machine` and choose **Ubuntu** as the operating system.
- Allow **HTTP traffic from the internet**.
- Click **Launch Instance**.
- Once the instance is launched, open the terminal and follow these steps:
  - Set the name of the runner as `self-hosted`.

### 4. **Configure GitHub Secrets and Variables 🔐:**

To configure the necessary secrets and variables for GitHub Actions, follow these steps:

1. Go to your **GitHub repository**.
2. Click on the **Settings** tab.
3. In the left sidebar, click on **Secrets and Variables** > **Actions**.
4. Click on **New repository secret** to add the following secrets:
   - `AWS_ACCESS_KEY_ID`: Add the Access Key ID from the IAM user `.csv` file.
   - `AWS_SECRET_ACCESS_KEY`: Add the Secret Access Key from the IAM user `.csv` file.
   - `AWS_REGION`: Set it to `us-east-1` or your desired AWS region.
   - `AWS_ECR_LOGIN_URI`: Add the ECR URI you copied (e.g., `demo566373416292.dkr.ecr.ap-south-1.amazonaws.com`).
   - `ECR_REPOSITORY_NAME`: Set it to the name of your ECR repository, e.g., `simple-app`.

5. After adding all the necessary secrets, click **Add Secret** to save each one.

### 5. **Configure EC2 Security Group 🔒:**

- Go to **EC2** > **Instances**, and click on the **Instance ID**.
- Under **Security**, click **Security Groups** and then **Edit Inbound Rules**.
- Add a new rule to allow **port 8080** and save the changes.

---

## After Execution: Clean Up 🧹

1. **Remove EC2 Instance 🔥**:
   - Go to **EC2** > **Instances**, select your instance, and click **Terminate**.

2. **Delete ECR Repository 🗑️**:
   - Go to **ECR**, select your repository, and click **Delete**.

3. **Delete IAM User 🗑️**:
   - Go to **IAM**, select `mlproj-user`, and click **Delete**.

---

By following these steps, we've set up a fully automated pipeline using GitHub Actions, AWS EC2, ECR, and IAM. This enables seamless deployment of machine learning models in a scalable and secure cloud environment. 🌍📈
