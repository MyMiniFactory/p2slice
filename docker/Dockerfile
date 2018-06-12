FROM centos:7

WORKDIR /app
ADD ./docker/requirements.txt /app
ADD ./upload_process_P2Slice /app

RUN rpm -Uvh https://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm
RUN rpm -Uvh https://mirror.webtatic.com/yum/el7/webtatic-release.rpm
# RUN yum -y install vim git curl

# Install Python 3.4 and extensions
RUN yum -y install python34.x86_64 \
                   python34-numpy.x86_64 \
                   python34-pip.noarch

# pip install necessary python packages
RUN python3 -m pip install --trusted-host pypi.python.org -r requirements.txt


