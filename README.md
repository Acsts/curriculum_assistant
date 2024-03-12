 # AI Curriculum assistant

 ## Project goal

 The goal of this project is to deploy an app integrating an AI assistant (using OpenAI API) that can answer to questions asked on a given context (in this example, my curriculum vitae), and only on this context. 

 ## Usage

1. You must have Docker and Docker compose installed on your computer
   
2. Clone this repository on your local machine
   
3. In the cloned repository, create a ".env" file and set inside it your OPENAI_API_KEY:

```.env
OPENAI_API_KEY="<YOUR API KEY>"
```

4. run this command:

```bash
docker compose up
```
And open the link given by the logs of the "front" container.

## Author

Antoine Costes
