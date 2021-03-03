FROM python:3.8
RUN mkdir /app
WORKDIR /app
ADD . /app/
RUN pip install -r requirements.txt
EXPOSE 6443
CMD ["python3", "main.py"]
