# Project Debug Rules (Non-Obvious Only)

- **Docker Logs**: Use `./scripts/launch.sh logs dev` to see combined logs; `docker logs agentic-workstation` for agent container.
- **GPU Debugging**: `nvidia-smi` inside container to verify GPU passthrough; `nvcc --version` to check CUDA.
- **Neo4j**: Accessible via `localhost:7474`; credentials in `docker-compose.unified.yml` or `.env`.
- **Rust Backtraces**: Set `RUST_BACKTRACE=1` in `.env` or shell for Actix errors.
