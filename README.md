# Use Docker Compose to build the project
```bash
docker compose up -d
```
## Note: Make sure to fill in the API keys in ```.env``` file. And the cloudflared tunnel token in docker-compose.yaml file.

### Access the Langgraph Studio locally through URL: 
https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
https://smith.langchain.com/studio/?baseUrl=http://0.0.0.0:2024
### Or access OpenWebUI interface through the domain name of the clouflared tunnel, or locally through http://localhost:8080
