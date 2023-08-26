
# Debian allows faster, smaller Docker builds than Alpine for Python
#   https://pythonspeed.com/articles/alpine-docker-python/
#   Using Alpine can make Python Docker builds 50× slower
#   by Itamar Turner-Trauring
#   Last updated 04 May 2022, originally created 29 Jan 2020

# Advice on best Python base image:
#   https://pythonspeed.com/articles/base-image-python-docker-images/
#   The best Docker base image for your Python application (Sep 2022)
#   by Itamar Turner-Trauring
#   Last updated 25 Oct 2022, originally created 30 Aug 2021

# Python 3.9 was chosen by Poetry for this app
FROM python:3.9

# Get updates and security fixes as root user before switching to non-root user
#   https://pythonspeed.com/articles/security-updates-in-docker/
#   The worst so-called “best practice” for Docker
#   by Itamar Turner-Trauring
#   Last updated 01 Oct 2021, originally created 23 Mar 2021
RUN apt-get update && apt-get -y upgrade


WORKDIR /src

# Poetry dependencies were exported to 'requirements.txt' 
COPY src/requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8050


# Switch to non-root user
RUN useradd --create-home appuser
USER appuser


# Should environment variables be set by root user or non-root user?
#   It seems like root user should set them, but app works either way, so apply
#   principle of least privilege and use non-root user for now
# Environment variables copied from:
#   https://stackoverflow.com/questions/53835198/integrating-python-poetry-with-docker
#   Integrating Python Poetry with Docker
ENV PYTHONFAULTHANDLER=1 \
  PYTHONHASHSEED=random \
  # "Seems to speed things up"
  #   https://pmac.io/2019/02/multi-stage-dockerfile-and-python-virtualenv/
  #   Multi-Stage Dockerfiles and Python Virtualenvs
  #   Mon, Feb 18, 2019
  PYTHONUNBUFFERED=1 \
  PIP_DEFAULT_TIMEOUT=100 \
  PIP_DISABLE_PIP_VERSION_CHECK=1 \
  PIP_NO_CACHE_DIR=1


CMD ["python", "src/dirichlet_multinomial.py"]
