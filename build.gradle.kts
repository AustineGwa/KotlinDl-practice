import org.jetbrains.kotlin.gradle.tasks.KotlinCompile

plugins {
    kotlin("jvm") version "1.4.10"
}

group = "me.gwaza"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
    jcenter()
    maven(url = "https://kotlin.bintray.com/kotlin-datascience")
}

dependencies {
    testImplementation(kotlin("test-junit"))
    implementation ("org.jetbrains.kotlin-deeplearning:api:0.1.1")
    //implementation ("org.slf4j:slf4j-log4j12:1.7.29")
}

tasks.test {
    useJUnit()
}

tasks.withType<KotlinCompile>() {
    kotlinOptions.jvmTarget = "1.8"
}