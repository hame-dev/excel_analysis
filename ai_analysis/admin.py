from django.contrib import admin

from ai_analysis.models import AgentTraceEvent, ChatMessage, ChatSession, RuntimeConfig, UploadRecord


@admin.register(UploadRecord)
class UploadRecordAdmin(admin.ModelAdmin):
    list_display = ("id", "original_filename", "is_active", "row_count", "column_count", "uploaded_at", "file_deleted")
    list_filter = ("is_active", "file_deleted", "uploaded_at")
    search_fields = ("original_filename", "file_hash")


@admin.register(ChatSession)
class ChatSessionAdmin(admin.ModelAdmin):
    list_display = ("id", "title", "updated_at", "created_at")
    search_fields = ("title",)


@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display = ("id", "session", "role", "status", "upload_record", "created_at")
    list_filter = ("role", "status", "created_at", "session")
    search_fields = ("content", "error_text")


@admin.register(AgentTraceEvent)
class AgentTraceEventAdmin(admin.ModelAdmin):
    list_display = ("id", "message", "step_index", "event_kind", "created_at")
    list_filter = ("event_kind", "created_at")


@admin.register(RuntimeConfig)
class RuntimeConfigAdmin(admin.ModelAdmin):
    list_display = ("id", "ollama_base_url", "ollama_qa_model", "ollama_chat_model", "ollama_embed_model", "verified_at", "updated_at")
    search_fields = ("ollama_base_url", "ollama_qa_model", "ollama_chat_model", "ollama_embed_model")
