FROM alpine:latest
RUN apk add --no-cache build-base git cmake python3
RUN python3 -m ensurepip
RUN pip3 install --no-cache --upgrade pip setuptools
RUN pip3 install flask

WORKDIR /
RUN touch README.md
COPY ./print_hello bin/print_hello
RUN chmod +x bin/print_hello

COPY . .
RUN mkdir build
WORKDIR /build
RUN cmake .. && make

RUN adduser -D devops2023user
USER devops2023user
WORKDIR /
ENTRYPOINT python3 app.py --host=0.0.0.0 --port=8898
