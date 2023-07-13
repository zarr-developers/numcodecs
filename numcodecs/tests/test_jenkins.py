import numpy
import pytest

from numcodecs.jenkins import jenkins_lookup3, JenkinsLookup3

def test_jenkins_lookup3():
    h = jenkins_lookup3(b"", 0)
    assert(h == 0xdeadbeef)
    h = jenkins_lookup3(b"", 0xdeadbeef)
    assert(h == 0xbd5b7dde)
    h = jenkins_lookup3(b"Four score and seven years ago", 0)
    assert(h == 0x17770551)
    h = jenkins_lookup3(b"Four score and seven years ago", 1)
    assert(h == 0xcd628161)
    # jenkins-cffi example
    h = jenkins_lookup3(b"jenkins", 0)
    assert(h == 202276345)
    h_last = 0
    h = jenkins_lookup3(b"", h_last)
    assert(h != h_last)
    h_last = h
    h = jenkins_lookup3(b"", h_last)
    assert(h != h_last)
    h_last = h
    h = jenkins_lookup3(b"", h_last)
    assert(h != h_last)
    h_last = h
    h = jenkins_lookup3(b"", h_last)
    assert(h != h_last)
    h_last = h
    h = jenkins_lookup3(b"", h_last)
    assert(h != h_last)
    h_last = h
    h = jenkins_lookup3(b"", h_last)
    assert(h != h_last)
    h_last = h
    h = jenkins_lookup3(b"", h_last)
    assert(h != h_last)
    h_last = h
    h = jenkins_lookup3(b"", h_last)
    assert(h != h_last)
    h_last = h
    h = jenkins_lookup3(b"", h_last)
    assert(h != h_last)

def test_jenkins_lookup3_codec():
    s = b"Four score and seven years ago"
    j = JenkinsLookup3()
    result = j.encode(s, 0)
    assert(result[-4:] == b'\x51\x05\x77\x17')
    assert(bytes(j.decode(result)) == s)
