from worker import process_dehazing_task
from flask import current_app
import os

def submit_dehazing_task(image_path, output_folder):
    """
    Submit dehazing task to Celery worker
    """
    try:
        # Submit task asynchronously
        task = process_dehazing_task.delay(image_path, output_folder)

        return {
            'task_id': task.id,
            'status': 'PENDING'
        }

    except Exception as e:
        return {
            'error': f"Failed to submit task: {str(e)}",
            'status': 'ERROR'
        }

def get_task_status(task_id):
    """
    Get the status of a Celery task
    """
    try:
        from worker import app as celery_app
        task_result = celery_app.AsyncResult(task_id)

        if task_result.state == 'PENDING':
            response = {
                'state': task_result.state,
                'status': 'Task is pending...'
            }
        elif task_result.state == 'PROGRESS':
            response = {
                'state': task_result.state,
                'progress': task_result.info.get('progress', 0),
                'status': task_result.info.get('status', 'Processing...')
            }
        elif task_result.state == 'SUCCESS':
            response = {
                'state': task_result.state,
                'result': task_result.result,
                'status': 'Task completed successfully'
            }
        else:
            response = {
                'state': task_result.state,
                'status': str(task_result.info)
            }

        return response

    except Exception as e:
        return {
            'state': 'ERROR',
            'status': f"Failed to get task status: {str(e)}"
        }
