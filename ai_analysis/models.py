from django.db import models
from django.utils import timezone


class UploadRecord(models.Model):
    NULL_STRATEGY_NONE = "none"
    NULL_STRATEGY_FRONT = "frontfill"
    NULL_STRATEGY_BACK = "backfill"
    NULL_STRATEGY_BOTH = "back_forward_fill"
    NULL_STRATEGY_MEAN = "mean_fill"
    NULL_STRATEGY_ZERO = "zero_fill"
    NULL_STRATEGY_CHOICES = [
        (NULL_STRATEGY_NONE, "No Fill"),
        (NULL_STRATEGY_FRONT, "Forward Fill"),
        (NULL_STRATEGY_BACK, "Backward Fill"),
        (NULL_STRATEGY_BOTH, "Forward + Backward Fill"),
        (NULL_STRATEGY_MEAN, "Mean Fill (numeric)"),
        (NULL_STRATEGY_ZERO, "Zero Fill"),
    ]

    original_filename = models.CharField(max_length=255)
    stored_file = models.FileField(upload_to="uploaded_excels/", null=True, blank=True)
    file_hash = models.CharField(max_length=64)
    row_count = models.PositiveIntegerField(default=0)
    column_count = models.PositiveIntegerField(default=0)
    columns_json = models.JSONField(default=list, blank=True)
    null_strategy = models.CharField(
        max_length=32,
        choices=NULL_STRATEGY_CHOICES,
        default=NULL_STRATEGY_NONE,
    )
    is_active = models.BooleanField(default=False, db_index=True)
    file_deleted = models.BooleanField(default=False)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    deactivated_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-uploaded_at"]

    def deactivate(self) -> None:
        self.is_active = False
        self.deactivated_at = timezone.now()
        self.save(update_fields=["is_active", "deactivated_at"])

    def __str__(self) -> str:
        status = "active" if self.is_active else "inactive"
        return f"{self.original_filename} ({status})"


class ChatSession(models.Model):
    title = models.CharField(max_length=120, default="New Chat")
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    updated_at = models.DateTimeField(auto_now=True, db_index=True)

    class Meta:
        ordering = ["-updated_at", "-id"]

    def __str__(self) -> str:
        return f"session#{self.id} {self.title}"


class RuntimeConfig(models.Model):
    ollama_base_url = models.CharField(max_length=255, default="")
    ollama_qa_model = models.CharField(max_length=120, default="")
    ollama_chat_model = models.CharField(max_length=120, default="")
    ollama_embed_model = models.CharField(max_length=120, default="")
    custom_prompt = models.TextField(blank=True, default="")
    verified_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True, db_index=True)

    class Meta:
        ordering = ["-updated_at", "-id"]

    def __str__(self) -> str:
        return f"runtime-config#{self.id}"


class ChatMessage(models.Model):
    ROLE_USER = "user"
    ROLE_ASSISTANT = "assistant"
    ROLE_CHOICES = [
        (ROLE_USER, "User"),
        (ROLE_ASSISTANT, "Assistant"),
    ]

    STATUS_PENDING = "pending"
    STATUS_RUNNING = "running"
    STATUS_DONE = "done"
    STATUS_ERROR = "error"
    STATUS_CHOICES = [
        (STATUS_PENDING, "Pending"),
        (STATUS_RUNNING, "Running"),
        (STATUS_DONE, "Done"),
        (STATUS_ERROR, "Error"),
    ]

    role = models.CharField(max_length=20, choices=ROLE_CHOICES)
    session = models.ForeignKey(
        ChatSession,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="messages",
    )
    content = models.TextField(blank=True, default="")
    citations = models.JSONField(default=list, blank=True)
    upload_record = models.ForeignKey(
        UploadRecord,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="messages",
    )
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default=STATUS_DONE, db_index=True)
    error_text = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["created_at", "id"]

    def __str__(self) -> str:
        return f"{self.role}#{self.id} ({self.status})"


class AgentTraceEvent(models.Model):
    EVENT_TRACE = "trace"
    EVENT_STATUS = "status"
    EVENT_CHOICES = [
        (EVENT_TRACE, "Trace"),
        (EVENT_STATUS, "Status"),
    ]

    message = models.ForeignKey(ChatMessage, on_delete=models.CASCADE, related_name="trace_events")
    step_index = models.PositiveIntegerField(default=0, db_index=True)
    event_kind = models.CharField(max_length=20, choices=EVENT_CHOICES, default=EVENT_TRACE)
    payload = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        ordering = ["id"]

    def __str__(self) -> str:
        return f"trace#{self.id} msg#{self.message_id} step={self.step_index}"
