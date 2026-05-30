# Automatic Mask Placement (AMP) GUI

The Automatic Mask Placement GUI is the **primary and recommended interface** for most users.
It provides visualization and interactive control, making it the most efficient way
to design, debug, and validate mask placement strategies.

The CLI interface is intended for batch processing and fully reproducible large-scale generation
after configurations have been verified through the GUI.

To start the GUI server:

```bash
AMP_PORT=5000 python3 -m amp_gui.backend.app
```

`AMP_PORT` specifies the port number on which the GUI backend server listens.
If not explicitly set, the backend will use its default port 5000.

After starting the server, open a web browser and navigate to:
```
http://localhost:<AMP_PORT>
```