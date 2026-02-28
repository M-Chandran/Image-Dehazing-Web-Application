from celery import Celery
from dehazing.dehaze import dehaze_image
import os

# Initialize Celery app
app = Celery('dehazing_worker',
             broker='redis://localhost:6379/0',
             backend='redis://localhost:6379/0')

app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
)

@app.task(bind=True)
def process_dehazing_task(self, image_path, output_folder):
    """
    Celery task for processing dehazing asynchronously
    """
    try:
        # Update task state to PROGRESS
        self.update_state(
            state='PROGRESS',
            meta={'progress': 10, 'status': 'Starting dehazing process...'}
        )

        # Validate input file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Input image not found: {image_path}")

        # Ensure output directory exists
        os.makedirs(output_folder, exist_ok=True)

        # Process the image
        self.update_state(
            state='PROGRESS',
            meta={'progress': 30, 'status': 'Loading AI models...'}
        )

        result = dehaze_image(image_path, output_folder)

        self.update_state(
            state='PROGRESS',
            meta={'progress': 90, 'status': 'Finalizing results...'}
        )

        # Return the result
        if isinstance(result, tuple):
            output_path, steps = result
        else:
            output_path = result
            steps = {}

        output_filename = os.path.basename(output_path)
        steps_filenames = {k: os.path.basename(v) for k, v in steps.items()}

        return {
            'output_image': output_filename,
            'steps': steps_filenames,
            'progress': 100,
            'status': 'Dehazing completed successfully'
        }

    except Exception as e:
        # Update task state to FAILURE
        self.update_state(
            state='FAILURE',
            meta={'error': str(e), 'status': 'Task failed'}
        )
        raise e

if __name__ == '__main__':
    app.start()
