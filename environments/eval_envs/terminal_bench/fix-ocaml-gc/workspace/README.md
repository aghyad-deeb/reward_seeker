This task's real working tree lives inside the task container at `/app/ocaml`.

Use the runtime helpers:
- `runtime/start_container.sh`
- `runtime/enter_container.sh`
- `runtime/run_eval.sh`
- `runtime/export_workspace.sh`

Notes:
- `instruction.md` is copied into the container as `/app/TASK.md`.
- A host-side copy of the codebase can be exported to `workspace/ocaml`, but the
  authoritative task environment is the live container.
- The reference solution is intentionally not copied into this local setup.
