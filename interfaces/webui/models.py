from .app import db # Import db instance from app
import datetime

class Task(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    celery_task_id = db.Column(db.String(128), unique=True, nullable=True) # Celery's internal ID
    status = db.Column(db.String(20), nullable=False, default='PENDING') # PENDING, RUNNING, COMPLETED, FAILED
    progress = db.Column(db.Integer, default=0)
    message = db.Column(db.String(256), nullable=True)
    start_time = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    end_time = db.Column(db.DateTime, nullable=True)
    result_path = db.Column(db.String(512), nullable=True) # Path to the results ZIP
    config_json = db.Column(db.Text, nullable=True) # Store config used for the task

    def __repr__(self):
        return f'<Task {self.id} ({self.status})>'

class LayerDefinition(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    layer_id_name = db.Column(db.String(80), unique=True, nullable=False) # Unique name like 'dense_via_grid'
    config_json = db.Column(db.Text, nullable=False) # JSON string of the layer parameters
    is_enabled = db.Column(db.Boolean, default=True)
    description = db.Column(db.String(256), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

    def __repr__(self):
        return f'<LayerDefinition {self.layer_id_name}>'
