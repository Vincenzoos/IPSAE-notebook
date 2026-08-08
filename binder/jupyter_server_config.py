# Raise the Jupyter websocket message limit so small FileUpload widgets can
# finish. AlphaFold Server zips are usually still larger — prefer the file
# browser + Extract zip path in the Bulk Evaluation UI.
c.ServerApp.tornado_settings = {
    "websocket_max_message_size": 32 * 1024 * 1024,
}
