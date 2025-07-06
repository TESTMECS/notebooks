serve:
  python3 -m http.server 8000 -d "./public/pages"
edit:
  uv run marimo edit --watch 
sandbox:
  uv run marimo edit --watch --sandbox 
