from django.urls import path

from ai_analysis import views

urlpatterns = [
    path("", views.home, name="home"),
    path("upload", views.upload_page, name="upload_page"),
    path("chat", views.chat_page, name="chat_page"),
    path("visualization", views.visualization_page, name="visualization_page"),
    path("settings", views.settings_page, name="settings_page"),
    path("api/upload/preview-nulls", views.api_upload_preview_nulls, name="api_upload_preview_nulls"),
    path("api/upload/status", views.api_upload_status, name="api_upload_status"),
    path("api/upload", views.api_upload, name="api_upload"),
    path("api/settings/verify", views.api_settings_verify, name="api_settings_verify"),
    path("api/settings/save", views.api_settings_save, name="api_settings_save"),
    path("api/chat/sessions", views.api_chat_sessions, name="api_chat_sessions"),
    path("api/chat/messages", views.api_chat_messages, name="api_chat_messages"),
    path("api/chat/stream/<int:message_id>", views.api_chat_stream, name="api_chat_stream"),
    path("api/data/filters", views.api_data_filters, name="api_data_filters"),
    path("api/data/query", views.api_data_query, name="api_data_query"),
]
