Deployment bundle for download: 
https://s3.us-east-2.amazonaws.com/digit-recognition-code/deployment_bundle.zip

The steps to run the code locally are:
1.  Create a new virtualenv on your mac. Lets assume the targetDirectory name is 'TestDeployment'.  Follow steps 1- 3 here:  https://www.tensorflow.org/install/install_mac#installing_with_virtualenv 
2.  Unzip the deployment_bundle into to a separate folder, say to newly created /dist_folder. (Save to an empty folder before opening).
3. (Cut and) Copy the /src and /build folders the TestDeployment folder.
4. Copy the rest of the folders under the /dist_folder to TestDeployment/lib/python2.7/site-packages/
5. Copy the attached png file to the TestDeployment folder.
6. (Optional, since the build folder contains model): From TestDeployment folder, run 'python src/model_creation.py'
7.Run 'python src/model_deployment.py' from the TestDeployment folder. (My code currently uses relative paths).

Note:  The inputs are hardcoded as I was trying to get a self-contained version running on AWS Lambda before accepting inputs. 


Update regd. AWS :
Subsequently, I attempted to bundle my code and put it up on AWS Lambda. Unfortunately, AWS Lambda places limits on the size of the deployment packages to 50MB, and the deployment zip including numpy and tensorflow was over 150MB : https://docs.aws.amazon.com/lambda/latest/dg/limits.html

Given this constraint, I have researched 2 solutions for enabling a fully cloud pipeline (which allows users to feed in an image and receive a response).

Solution 1: Slim down the size of the modules (Link1) 
Solution 2: Use Docker/EC2 instances to run code. In this case, we'll need to use a python webserver (SimpleHTTPserver or Tornado) to accept web requests.   

Followed the first solution a little further - used modulefinder to find relevant modules (sorted_modules.txt attached). But if the goal is to have an extensible, reusable pipline, the second solution would be a better fit. 
