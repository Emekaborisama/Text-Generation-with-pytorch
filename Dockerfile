FROM python:3-slim

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN ls

COPY api.py api.py

EXPOSE 8000

CMD ["python", "api.py" ] 
CMD["ls"]