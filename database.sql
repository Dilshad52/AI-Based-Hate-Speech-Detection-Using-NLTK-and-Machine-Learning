DROP DATABASE IF EXISTS hate_speech_ai;

CREATE DATABASE hate_speech_ai;
USE hate_speech_ai;

CREATE TABLE users(
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(100) UNIQUE,
    password VARCHAR(100),
    email VARCHAR(150),
    mobile VARCHAR(20),
    address TEXT,

    status VARCHAR(20) DEFAULT 'active',
    warning_count INT DEFAULT 0,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE history(
    id INT AUTO_INCREMENT PRIMARY KEY,

    username VARCHAR(100),

    text TEXT,

    prediction VARCHAR(50),

    admin_action VARCHAR(50) DEFAULT 'pending',

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


CREATE TABLE admin(
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50),
    password VARCHAR(50)
);

INSERT INTO admin(username,password)
VALUES('admin','admin');
