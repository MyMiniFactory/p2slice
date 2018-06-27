FROM centos:7


RUN rpm -Uvh https://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm
RUN rpm -Uvh https://mirror.webtatic.com/yum/el7/webtatic-release.rpm
RUN yum -y install curl

# Install Python 3.4 and extensions
RUN yum -y install python34.x86_64 \
                   python34-numpy.x86_64 \
                   python34-pip.noarch

# download blender
RUN yum -y install bzip2 \
                    mesa-libGLU.x86_64 \
                    libXtst \
                    libXrender

ENV BLENDER_MAJOR 2.79
ENV BLENDER_VERSION 2.79
ENV BLENDER_BZ2_URL https://mirror.clarkson.edu/blender/release/Blender$BLENDER_MAJOR/blender-$BLENDER_VERSION-linux-glibc219-x86_64.tar.bz2

RUN mkdir /usr/local/blender
RUN curl -SL "$BLENDER_BZ2_URL" -o blender.tar.bz2
RUN tar -jxvf blender.tar.bz2 -C /usr/local/blender --strip-components=1
RUN rm blender.tar.bz2

ENV PATH="/usr/local/blender:${PATH}"

# pip install necessary python packages
WORKDIR /app
ADD ./docker/requirements.txt /app
ADD ./upload_process_P2Slice /app


RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install --trusted-host pypi.python.org -r requirements.txt

ENTRYPOINT ["python3", "upload_process_P2Slice.py"]
