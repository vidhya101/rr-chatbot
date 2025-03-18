from api.routes import chat, data_viz_routes

# Include routers
app.include_router(chat.router, prefix="/api", tags=["chat"])
app.include_router(data_viz_routes.router, prefix="/api/data", tags=["data-visualization"]) 