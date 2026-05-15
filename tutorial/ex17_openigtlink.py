"""
Ex 17 — OpenIGTLink live streamer.

SKIPPED: requires a running OpenIGTLink server on localhost:18944.
Typically used with Slicer 3D or a US machine sending real-time data.

To use when a server is available:
    streamer = fast.OpenIGTLinkStreamer.create("localhost", 18944)
    fast.display2D(streamer)
"""
print(__doc__)
