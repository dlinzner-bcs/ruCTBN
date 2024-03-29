# syntax=docker/dockerfile:1.0.0-experimental
# http://blog.oddbit.com/post/2019-02-24-docker-build-learns-about-secr/

FROM rust:buster as builder

RUN apt-get update && \
	apt-get upgrade -yy && \
	apt-get install -y \
			cmake \
			libssl-dev \
	&&\
	rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/ctbn
COPY . .
RUN rustup component add rustfmt
RUN --mount=type=ssh cargo build

FROM debian:buster-slim

RUN apt-get update && \
		apt-get install -y ca-certificates && \
		rm -rf /var/lib/apt/lists/*

EXPOSE 8000 50051


