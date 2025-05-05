from celery import Celery
import os

# Set default Redis URL; can be overridden by environment variable
REDIS_URL = os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379/0')

# Configure Celery instance
celery = Celery(
    'sem_generator_tasks', # Name of the celery app
    broker=REDIS_URL,
    backend=REDIS_URL, # Using Redis as result backend too
    include=['tasks'] # List of modules containing tasks to import
)

# Optional Celery configuration
celery.conf.update(
    result_expires=3600, # Keep results for 1 hour
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
    timezone='UTC',
    enable_utc=True,
    # Add task tracking settings
    task_track_started=True,
    # Add rate limits if needed
    # task_annotations = {'tasks.run_generation_task': {'rate_limit': '10/m'}}
)

if __name__ == '__main__':
    celery.start() # For running worker from command line if needed (see step 9)
