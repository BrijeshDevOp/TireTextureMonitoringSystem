# Deploying TireTextureApp to Render

## Prerequisites

- A GitHub account with your project pushed to a repository
- A Render account (free tier available)

## Step-by-Step Deployment Guide

### 1. Prepare Your Repository

Make sure your repository contains these essential files:

- `app.py` - Your Flask application
- `requirements.txt` - Python dependencies
- `runtime.txt` - Python version specification
- `model3.hdf5` - Your trained model
- `templates/` - HTML templates
- `static/` - Static assets (if any)

### 2. Deploy to Render

1. **Sign up/Login to Render**

   - Go to [render.com](https://render.com)
   - Sign up with your GitHub account

2. **Create a New Web Service**

   - Click "New +" button
   - Select "Web Service"
   - Connect your GitHub repository

3. **Configure the Web Service**

   - **Name**: `tire-texture-app` (or your preferred name)
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Plan**: Free (or choose paid plan for better performance)

4. **Environment Variables** (Optional)

   - Add any environment variables if needed
   - For this app, no additional environment variables are required

5. **Deploy**
   - Click "Create Web Service"
   - Render will automatically build and deploy your application
   - The first deployment may take 5-10 minutes due to TensorFlow installation

### 3. Important Notes

- **Model File Size**: Your `model3.hdf5` file is ~7.8MB, which is within Render's limits
- **Memory Usage**: TensorFlow can be memory-intensive. The free tier has 512MB RAM limit
- **Cold Starts**: The first request after inactivity may be slow as the model loads
- **Auto-Deploy**: Render automatically redeploys when you push changes to your main branch

### 4. Troubleshooting

**Common Issues:**

- **Build Failures**: Check that all dependencies are in `requirements.txt`
- **Memory Errors**: Consider upgrading to a paid plan for more RAM
- **Model Loading Issues**: Ensure `model3.hdf5` is in the root directory

**Logs**: Check the logs in Render dashboard for detailed error messages

### 5. Performance Optimization

For better performance on Render:

- Consider using a lighter model format (TensorFlow Lite)
- Implement model caching strategies
- Use a paid plan for better resources

## Your App URL

Once deployed, your app will be available at:
`https://your-app-name.onrender.com`

Replace `your-app-name` with the name you chose during setup.
