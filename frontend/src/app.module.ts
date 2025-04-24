import { BrowserModule } from '@angular/platform-browser';
import { NgModule } from '@angular/core';
import { HttpClientModule } from '@angular/common/http'; // Import HttpClientModule

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { ScrapingComponent } from './scraping/scraping.component';
import { AnalisisComponent } from './analisis/analisis.component';
import { ChatComponent } from './chat/chat.component';
import { SharedModule } from './shared/shared.module';

@NgModule({
  declarations: [
    AppComponent,
    ScrapingComponent,
    AnalisisComponent,
    ChatComponent
  ],
  imports: [
    BrowserModule,
    AppRoutingModule,
    HttpClientModule, // Add HttpClientModule to imports
    SharedModule
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }