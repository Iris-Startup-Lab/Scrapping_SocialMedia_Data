/// Recursión extra en el component o en el dataservice, checar 
//import { Component } from '@angular/core';
import { Component, OnInit } from '@angular/core';
import { DataService } from '../shared/data.service';
//import { NgFor } from '@angular/common';
import { CommonModule } from '@angular/common';
//import { HttpClientModule } from '@angular/common/http';
//import { HttpClient } from '@angular/common/http';

console.log('ScrapingComponent loaded'); 

// The next component called "ScrapingComponent" is the main component of the application.
// It is responsible for displaying the tweets obtained from the API and managing the data service.
@Component({
  selector: 'app-scraping',
  templateUrl: './scraping.component.html',
  //styleUrls: ['./scraping.component.css'],
  //imports: [],  
  //imports: [NgFor, CommonModule, HttpClientModule],
  //imports: [CommonModule, HttpClientModule],
  imports: [CommonModule],
  standalone: true,
  styles: ``
})

export class ScrapingComponent implements OnInit {
  tweets: any[] = [];
  sentimentResults: {[tweetId:string]: any}={};
  constructor(private dataService: DataService) { }
  
  ngOnInit(): void {
    console.log('ScrapingComponent initialized');
    this.dataService.getTweets().subscribe(
      (data) => {
        this.tweets = data;
        console.log('LLamando a la api para obtener Tweets recibidos:', this.tweets); 
        this.getSentimentForTweets()
      },
      (error) => {
        console.error('Error al obtener los tweets:', error);
      }
    );
  } 

  getSentimentForTweets(): void {
    const tweetIds = this.tweets.map(tweet => tweet.id);
    this.dataService.getSentimentResults(tweetIds).subscribe(
      (sentimentData) => {
        sentimentData.forEach(result => {
          console.log('Resultado del sentimiento:', result);
          this.sentimentResults[result.id] = result; // Asumiendo que el backend devuelve 'tweet' como el tweet_id
        });
        console.log('Resultados del análisis de sentimientos recibidos:', this.sentimentResults);
      },
      (error) => {
        console.error('Error al obtener los resultados del análisis de sentimientos:', error);
      }
    );
  }
}