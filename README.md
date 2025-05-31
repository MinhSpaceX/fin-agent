## Getting Started with Docker Compose

Use Docker Compose to build and run the project:

```bash
docker compose up -d
```

---

## 🛠️ Configuration Required

- **API Keys**  
  Make sure to **fill in the API keys** in the `.env` file before starting the containers.

- **Cloudflared Tunnel Token**  
  Add your **Cloudflared tunnel token** in the `docker-compose.yaml` file under the appropriate service configuration.

---

## Interfaces Available

### LangGraph Studio

Access LangGraph Studio locally via:

- [https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024](https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024)
- [https://smith.langchain.com/studio/?baseUrl=http://0.0.0.0:2024](https://smith.langchain.com/studio/?baseUrl=http://0.0.0.0:2024)

> Open it in LangChain's hosted UI using the links above.

---

### OpenWebUI

Access the OpenWebUI interface:

- Locally: [http://localhost:8080](http://localhost:8080)
- Or via your **Cloudflared tunnel's domain name**
