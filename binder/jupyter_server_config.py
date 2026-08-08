# Prefer JupyterLab Contents API (file browser) for full AF3 export zips.
# Widget FileUpload still uses websockets; raise the limit so medium archives
# can finish, but large multi-job folders should use Zip path / Extract zip.
c.ServerApp.tornado_settings = {
    "websocket_max_message_size": 256 * 1024 * 1024,
}
