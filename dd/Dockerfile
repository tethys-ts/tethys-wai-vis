# FROM tiangolo/uwsgi-nginx-flask:python3.8
# FROM tiangolo/meinheld-gunicorn-flask:python3.8
FROM tethysts/tethys-dash-base:1.1

# RUN apt-get update && apt-get install -y libspatialindex-dev python-rtree

# COPY requirements.txt ./

# RUN pip install --no-cache-dir -r requirements.txt

# RUN apt-get update && apt-get install -y nano
# Copy hello-cron file to the cron.d directory
# COPY tethys-cron /etc/cron.d/tethys-cron

# Give execution rights on the cron job
# RUN chmod 0644 /etc/cron.d/tethys-cron
# RUN echo "uwsgi_read_timeout 300s;" > /etc/nginx/conf.d/custom_timeout.conf

# Apply cron job
# RUN crontab /etc/cron.d/tethys-cron

COPY ./app /app

# CMD ["uvicorn", "app.main:server", "--proxy-headers", "--host", "0.0.0.0", "--port", "80"]
CMD ["gunicorn", "--conf", "app/gunicorn_conf.py", "--bind", "0.0.0.0:80", "app.main:server"]
