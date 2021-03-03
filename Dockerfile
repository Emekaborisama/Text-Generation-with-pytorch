FROM python:3.6-slim
RUN mkdir /app
WORKDIR /app
ADD . /app/
RUN pip install -r requirements.txt
EXPOSE 6443
CMD ["python", "api.py"]